"""End-to-end contract test for the draw session, offline.

Plays the executor's part with the executor's own code: the Python stages
produce orbit.csv and path.csv from a synthetic contact at the carriage-IK
witness pose, synthetic D405 captures are rendered through the URDF camera
frames from a plane through the contact, and `path_plan_check` (the C++
parser + planner, cpp/teleop) must accept both files from the poses the
executor would plan from. Skips the C++ half when the tool is not built.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "lib"))
sys.path.insert(0, str(REPO / "scripts" / "vision"))

import draw_kinematics as dk  # noqa: E402
import draw_path as dp  # noqa: E402
import draw_stage  # noqa: E402

WITNESS_JOINTS = np.array([
    0.173762112856, 1.544403791428, 0.826848268509,
    -0.061226826161, 0.121118485928, 1.642061471939])
CARRIAGE = dk.CARRIAGE_IK_BIAS_M
PERIOD = 0.0025
PLAN_CHECK = REPO / "cpp" / "teleop" / "build" / "path_plan_check"
INTRINSICS = np.array([430.0, 430.0, 320.0, 240.0, 640.0, 480.0])


def _pose_json(joints, carriage):
    tip, rotation, _ = dk.fk_ballpoint(joints, carriage)
    return {"schema": "tatbot.draw-pose/1", "frame": "right/base_link", "period_s": PERIOD,
            "joints": [float(v) for v in joints], "carriage_m": float(carriage),
            "tip": tip.tolist(), "rotation": rotation.tolist(),
            "tool": "lutin-ballpoint-dot", "t_wall": 0.0}


def _stage(stage, draw_dir):
    return subprocess.run([sys.executable, str(REPO / "scripts" / "draw_stage.py"), stage, str(draw_dir)],
                          capture_output=True, text=True, timeout=600)


def _render_plane_depth(root_from_optical, point_root, normal_root, rng):
    fx, fy, ppx, ppy, width, height = INTRINSICS
    cols, rows = np.meshgrid(np.arange(int(width)), np.arange(int(height)))
    rays = np.stack([(cols - ppx) / fx, (rows - ppy) / fy, np.ones_like(cols, float)], axis=-1)
    d = rays.reshape(-1, 3) @ root_from_optical[:3, :3].T
    o = root_from_optical[:3, 3]
    denominator = d @ normal_root
    t = np.full(len(d), np.nan)
    ok = np.abs(denominator) > 1e-9
    t[ok] = ((point_root - o) @ normal_root) / denominator[ok]
    t += rng.normal(0.0, 0.0003, size=t.shape)
    depth = np.zeros(len(d), np.uint16)
    good = np.isfinite(t) & (t > 0.07) & (t < 0.5)
    depth[good] = np.round(t[good] / 1e-4).astype(np.uint16)
    return depth.reshape(int(height), int(width))


def _plan_check(samples_csv, joints, carriage):
    argv = [str(PLAN_CHECK), str(samples_csv), str(PERIOD), *[f"{v:.12f}" for v in joints], f"{carriage:.9f}"]
    return subprocess.run(argv, capture_output=True, text=True, timeout=120)


def test_orbit_map_and_path_are_accepted_by_the_executor(tmp_path):
    draw_dir = tmp_path / "draw"
    (draw_dir / "capture").mkdir(parents=True)
    config = dict(draw_stage.DEFAULT_CONFIG)
    config.update({"python": sys.executable, "rerun_connect": "", "scan_only": False})
    # The witness pose is the paper A/B's wrapped-wrist contact: the elbow sits near
    # its limit, and no off-axis angle keeps the default 15 mm design's 18 mm ring in
    # both frustums past the joint planner from there (docs/draw.md). The pipeline is
    # exercised on the 6 mm design that pose can see.
    config["design"] = {"kind": "spiral", "radius_mm": 6, "turns": 3, "rotation_deg": 0}
    (draw_dir / "draw.json").write_text(json.dumps(config))
    trigger = _pose_json(WITNESS_JOINTS, CARRIAGE)
    (draw_dir / "trigger.json").write_text(json.dumps(trigger))

    orbit = _stage("orbit", draw_dir)
    assert orbit.returncode == 0, orbit.stdout + orbit.stderr
    samples, header = dp.read_samples_csv(draw_dir / "orbit.csv")
    assert header["kind"] == "orbit" and int(header["capture_count"]) == 5

    # Joints along the orbit, from the advisory planner with the pen-up carriage lock.
    plan = dk.plan_joints(samples, WITNESS_JOINTS, CARRIAGE, PERIOD, lock_carriage_when_up=True)
    positions = plan["positions"]
    chain = dk.urdf_chain()
    axis = dk.tool_axis_in_link6()
    contact_root = dk.root_from_base(np.asarray(trigger["tip"]))
    normal_root = -(np.asarray(trigger["rotation"]) @ axis)
    rng = np.random.default_rng(0)
    capture_rows = np.nonzero(samples.capture > 0)[0]
    assert len(capture_rows) == 5
    for row in capture_rows:
        k = int(samples.capture[row])
        joints, carriage = positions[row, :6], positions[row, 6]
        values = dk.joint_map(joints, carriage)
        payload = {"joints": joints, "carriage_m": carriage, "k": k, "t_wall": 0.0}
        for role, link in draw_stage.CAMERA_LINKS.items():
            pose = chain.link_pose(link, values)
            depth = _render_plane_depth(pose, contact_root, normal_root, rng)
            payload[f"depth_{role}"] = depth
            payload[f"valid_{role}"] = (depth > 0).astype(np.uint8) * 8
            payload[f"units_m_{role}"] = 1e-4
            payload[f"intrinsics_{role}"] = INTRINSICS
        np.savez(draw_dir / "capture" / f"capture-{k}.npz", **payload)
        (draw_dir / "capture" / f"capture-{k}.done").touch()
    hold_joints, hold_carriage = positions[-1, :6], positions[-1, 6]
    (draw_dir / "hold.json").write_text(json.dumps(_pose_json(hold_joints, hold_carriage)))

    mapped = _stage("map", draw_dir)
    assert mapped.returncode == 0, mapped.stdout + mapped.stderr
    assert (draw_dir / "surface.npz").exists() and (draw_dir / "path.csv").exists()
    preflight = json.loads((draw_dir / "preflight.json").read_text())
    assert preflight.get("lean_max_deg", 0.0) < 1.0
    assert abs(preflight.get("arc_length_ratio", 1.0) - 1.0) < 0.01
    fit = json.loads((draw_dir / "surface.json").read_text())
    assert fit["chart_kind"] == "plane", fit

    if not (PLAN_CHECK.exists() and shutil.which(str(PLAN_CHECK))):
        pytest.skip("cpp/teleop/build/path_plan_check is not built here")
    orbit_check = _plan_check(draw_dir / "orbit.csv", WITNESS_JOINTS, CARRIAGE)
    assert orbit_check.returncode == 0, orbit_check.stderr
    path_check = _plan_check(draw_dir / "path.csv", hold_joints, hold_carriage)
    assert path_check.returncode == 0, path_check.stderr
    report = dict(line.split(",", 1) for line in path_check.stdout.strip().splitlines())
    assert float(report["model_max_error_mm"]) < 0.1
    assert float(report["plan_min_carriage_mm"]) >= dk.CARRIAGE_IK_MIN_M * 1e3
    assert float(report["plan_max_carriage_mm"]) <= dk.CARRIAGE_IK_MAX_M * 1e3


def test_mask_tool_drops_the_pen_body_but_keeps_the_surface():
    tip = np.array([0.3, -0.25, 0.16])
    axis = np.array([1.0, 0.0, 0.0])
    rng = np.random.default_rng(1)
    # pen: a cone from 1 mm radius at the tip widening 0.4 mm/mm to the 60 mm-back body; surface:
    # 2 mm past the tip, a 60 mm square
    t = rng.uniform(-0.06, -0.001, 500)
    ang = rng.uniform(0, 2 * np.pi, 500)
    radius = np.minimum(0.015, 0.001 + 0.4 * (-t))
    pen = tip + t[:, None] * axis + radius[:, None] * np.stack([np.zeros(500), np.cos(ang), np.sin(ang)], 1)
    yz = rng.uniform(-0.03, 0.03, (500, 2))
    surface = tip + 0.002 * axis + np.stack([np.zeros(500), yz[:, 0], yz[:, 1]], 1)
    kept = draw_stage.mask_tool(np.concatenate([pen, surface]), tip, axis, 0.02)
    assert len(kept) == 500
    assert np.allclose(kept, surface)
    # Leaning 37 deg off the normal, surface points to the side of the contact are behind the tip
    # along the axis but outside the pen's cone; they must survive (the live bottle capture did not).
    lean = np.deg2rad(37.0)
    tilted = np.array([np.cos(lean), np.sin(lean), 0.0])
    kept = draw_stage.mask_tool(np.concatenate([pen, surface]), tip, tilted, 0.02)
    surface_kept = kept[np.isclose(kept[:, 0], tip[0] + 0.002)]
    assert len(surface_kept) >= 480, len(surface_kept)
