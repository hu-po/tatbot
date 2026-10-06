"""The pinocchio CLIK against the C++ planner it was ported from (cpp/teleop path_plan_check).

Identical Cartesian samples (a square, a circle, a jellyfish-like pen-up + pen-down stroke, and a small
circle through the carriage-in-the-solve branch) go through both; the joint plans must agree at the tip by FK within motion.yaml clik.model_error_cap_m (0.1 mm),
and joint by joint within the same cap over the arm's reach. SKIPs when the binary is absent: build it
with `cmake -B cpp/teleop/build -S cpp/teleop -DCMAKE_BUILD_TYPE=Release && cmake --build
cpp/teleop/build --target path_plan_check`, or point TATBOT_PATH_PLAN_CHECK at one.
"""
import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import tatbot_motion as tm
from tatbot_description import repo_root
from tatbot_motion import timelaw as tl
from tatbot_motion.clik import clik

CPP_BALLPOINT_TIP_IN_LINK6 = (0.20498692817078468, 0.012312678895000949, -0.0005439999999999881)
REACH_M = 0.45  # a joint error of e rad moves the tip at most e * reach


def _binary() -> Path | None:
    env = os.environ.get("TATBOT_PATH_PLAN_CHECK")
    if env:
        return Path(env)
    try:
        path = repo_root() / "cpp" / "teleop" / "build" / "path_plan_check"
    except FileNotFoundError:
        return None
    return path if path.is_file() else None


@pytest.fixture(scope="module")
def setup():
    binary = _binary()
    if binary is None or not binary.is_file():
        pytest.skip("cpp/teleop/build/path_plan_check is not built")
    import pinocchio as pin

    kin = tm.Kinematics.from_repo(arm="right")
    motion = tm.load_motion()
    pin.framesForwardKinematics(kin.model, kin.data, pin.neutral(kin.model))
    link6 = kin.data.oMf[kin.model.getFrameId("right/link_6")]
    tcp = kin.data.oMf[kin.model.getFrameId("right/tcp")]
    link6_from_tcp = link6.inverse() * tcp
    tip = np.array(link6_from_tcp.translation)
    if np.linalg.norm(tip - CPP_BALLPOINT_TIP_IN_LINK6) > 1e-4:
        pytest.skip("config/workspace.yaml's TCP differs from the C++ ballpoint constant; rebuild the C++ first")
    return binary, kin, motion, np.array(link6_from_tcp.rotation), tip


def _seed(kin, height_m):
    """A drawing pose: tip over a fixed point (the old rig's page, not stack.yaml's), tcp z straight down."""
    q = np.array([-0.5, 1.0, 1.0, -0.3, 0.0, 0.0, 0.002])
    start = kin.fk(q)
    target = np.eye(4)
    target[:3, :3] = tl.align_rotation(start[:3, 2], [0.0, 0.0, -1.0]) @ start[:3, :3]
    target[:3, 3] = [0.371, -0.231, 0.0334 + height_m]
    return kin.solve(target, q)


def _run_cpp(binary, tmp_path, name, q0, p, v, r_link6, pen, carriage_ik, dt, tip):
    path = tmp_path / f"{name}.csv"
    lines = ["schema,tatbot.draw-samples/1", "kind,parity", "frame,right/base_link", "arm,right",
             f"period_s,{dt!r}", f"tip_x_m,{float(tip[0])!r}", f"tip_y_m,{float(tip[1])!r}", f"tip_z_m,{float(tip[2])!r}",
             f"sample_count,{len(p)}", "start_tolerance_m,0.001", f"carriage_ik,{int(carriage_ik)}",
             "columns,t_s,px,py,pz,vx,vy,vz,r00,r01,r02,r10,r11,r12,r20,r21,r22,pen,capture"]
    for k in range(len(p)):
        values = [k * dt, *p[k], *v[k], *r_link6[k].reshape(-1)]
        lines.append(",".join(repr(float(x)) for x in values) + f",{int(pen[k])},0")
    path.write_text("\n".join(lines) + "\n")
    out = subprocess.run([str(binary), str(path), repr(dt), *[repr(float(x)) for x in q0], "--json"],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def _square(tip0, rot, dt):
    corners = [tip0 + d for d in ([0.02, 0, 0], [0.02, 0.02, 0], [0, 0.02, 0], [0, 0, 0])]
    parts, here = [], tip0
    for corner in corners:
        p, _ = tl.line_segment(here, np.asarray(corner), rot, rot, 0.010, dt)
        parts.append(p)
        here = np.asarray(corner)
    p = np.concatenate([tip0[None]] + parts)
    return p, np.ones(len(p), bool)


def _circle(tip0, rot, dt, radius=0.008, speed=0.005):
    length = 2 * math.pi * radius
    _, s, _ = tl.time_law(length, length / speed + 1.0, 1.0, dt)
    angle = s / radius
    center = tip0 - [radius, 0.0, 0.0]
    p = center + np.stack([radius * np.cos(angle), radius * np.sin(angle), np.zeros_like(angle)], axis=1)
    p = np.concatenate([tip0[None], p])
    return p, np.ones(len(p), bool)


def _jellyfish(tip0, rot, dt, motion):
    """Lift, pen-up travel and descent to a wavy tentacle with two sharp corners, drawn pen-down."""
    x = np.linspace(0.0, 0.03, 300)
    tentacle = np.stack([x, 0.004 * np.sin(x / 0.03 * 3 * math.pi), np.zeros_like(x)], axis=1)
    tentacle = np.concatenate([tentacle, tentacle[-1] + [[0.0, -0.006, 0.0], [-0.008, -0.010, 0.0]]])
    start = tip0 + [0.005, 0.004, 0.0]
    poly = start + tentacle
    _, retime = tm.plan._cfg(motion)
    up, _, _ = tl.pen_up_leg(np.stack([tip0, tip0 + [0, 0, 0.01]]), rot, rot, [0.01], retime, dt, gentle_start=True)
    travel, _, _ = tl.pen_up_leg(np.stack([up[-1], start + [0, 0, 0.01], start + [0, 0, 0.005], start]),
                                 rot, rot, [0.12, 0.01, 0.003], retime, dt, gentle_end=True)
    settle, _ = tl.hold_rows(start, rot, 1.0, dt)
    _, s, _, _ = tl.corner_time_law(poly, 0.0035, 1.0, dt, retime)
    stroke, _ = tl.resample_polyline_by_arclength(poly, s)
    pen_up = np.concatenate([tip0[None], up, travel, settle])
    return np.concatenate([pen_up, stroke]), np.concatenate([np.zeros(len(pen_up), bool), np.ones(len(stroke), bool)])


@pytest.mark.parametrize("shape,carriage_ik", [("square", False), ("circle", False), ("jellyfish", False),
                                               ("small-circle", True)])
def test_clik_matches_path_plan_check(setup, tmp_path, shape, carriage_ik):
    binary, kin, motion, link6_from_tcp_rot, tip = setup
    dt = 1.0 / motion["control_rate_hz"]
    q0 = _seed(kin, 0.0)
    seed = kin.fk(q0)
    rot = seed[:3, :3]
    if shape == "square":
        p, pen = _square(seed[:3, 3], rot, dt)
    elif shape == "circle":
        p, pen = _circle(seed[:3, 3], rot, dt)
    elif shape == "small-circle":  # the weighted seven-joint solve keeps its 0.5-3.5 mm window only on small motion
        p, pen = _circle(seed[:3, 3], rot, dt, radius=0.0015, speed=0.002)
    else:
        p, pen = _jellyfish(seed[:3, 3], rot, dt, motion)
    v = tl.feedforward(p, dt)
    rots = np.repeat(rot[None], len(p), axis=0)
    cpp = _run_cpp(binary, tmp_path, f"{shape}-{int(carriage_ik)}", q0, p, v, rots @ link6_from_tcp_rot.T, pen,
                   carriage_ik, dt, tip)
    ours = clik(kin, q0, p, v, rots, pen, dt, motion["clik"], motion["carriage"], carriage_ik=carriage_ik)
    theirs = np.asarray(cpp["positions"])
    assert theirs.shape == ours.q.shape
    cap = motion["clik"]["model_error_cap_m"]
    tip_theirs = np.array([kin.fk(q)[:3, 3] for q in theirs])
    tip_gap = np.linalg.norm(tip_theirs - ours.tip, axis=1).max()
    joint_gap = np.abs(theirs[:, :6] - ours.q[:, :6]).max()
    carriage_gap = np.abs(theirs[:, 6] - ours.q[:, 6]).max()
    print(f"{shape} carriage_ik={carriage_ik}: {len(p)} samples, tip {tip_gap * 1e3:.2e} mm, joints {joint_gap:.2e} rad, "
          f"carriage {carriage_gap * 1e3:.2e} mm, C++ model error {cpp['max_model_error_mm']:.4f} mm, "
          f"ours {ours.model_error_m.max() * 1e3:.4f} mm")
    assert tip_gap <= cap
    assert joint_gap * REACH_M <= cap and carriage_gap <= cap
    # both track their own reference equally well
    assert abs(cpp["max_model_error_mm"] * 1e-3 - ours.model_error_m.max()) <= cap
    np.testing.assert_allclose(np.asarray(cpp["velocities"]), ours.qd, atol=1e-4)
