"""Pin scripts/lib/draw_path.py: time law, spiral, surface lift, preflight, files, orbit.

    uvx --with-requirements scripts/tests/requirements.txt pytest -q scripts/tests/test_draw_path.py

The surface here is a small local fake of the draw_surface.HeightFieldSurface
API (frame / project / count / width_m / height_m / chart / anchor_to) — a plane
and a 40 mm cylinder with closed-form geometry — so this module pins the path
compiler independently of the mapper.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "lib"))
sys.path.insert(0, str(REPO / "scripts" / "vision"))

import draw_kinematics as dk  # noqa: E402
import draw_path as dp  # noqa: E402

PERIOD = 0.0025
TOOL_AXIS = np.array([math.sqrt(0.5), -math.sqrt(0.5), 0.0])
CONFIG = {
    "schema": "tatbot.draw-config/1", "tool": "lutin-ballpoint-dot",
    "design": {"kind": "spiral", "radius_mm": 6, "turns": 3, "rotation_deg": 0},
    "duration_s": 30, "ease_s": 2, "scan_only": False,
    "orbit": {"standoff_mm": 120, "tilt_deg": 15, "poses": 5, "speed_mm_s": 10},
    "map": {"cell_mm": 1.0, "extent_mm": 60, "chart": "auto"},
    "lean_budget_deg": 20,
}


class FakePlane:
    """Plane chart: point = c + u e_u + v e_v, normal n. Cells all filled unless told otherwise."""

    def __init__(self, center, e_u, e_v, width_m=0.06, height_m=0.06, cell_m=0.001):
        self.center = np.asarray(center, float)
        self.e_u = np.asarray(e_u, float)
        self.e_v = np.asarray(e_v, float)
        self.n = np.cross(self.e_u, self.e_v)
        self.width_m = width_m
        self.height_m = height_m
        rows = int(round(height_m / cell_m)) + 1
        cols = int(round(width_m / cell_m)) + 1
        self.count = np.ones((rows, cols), dtype=np.int32)
        self.chart = SimpleNamespace(kind="plane", radius_m=float("nan"))

    def frame(self, uv):
        uv = np.asarray(uv, float)
        point = self.center + uv[:, :1] * self.e_u + uv[:, 1:] * self.e_v
        n = np.repeat(self.n[None], len(uv), axis=0)
        return point, np.repeat(self.e_u[None], len(uv), 0), np.repeat(self.e_v[None], len(uv), 0), n

    def project(self, points):
        q = np.asarray(points, float) - self.center
        return np.stack([q @ self.e_u, q @ self.e_v], axis=1), q @ self.n

    def anchor_to(self, point):
        uv, dist = self.project(np.asarray(point, float)[None])
        shifted = FakePlane(self.center + dist[0] * self.n, self.e_u, self.e_v, self.width_m, self.height_m)
        shifted.count = self.count
        return shifted, float(dist[0]), uv[0]


class FakeCylinder:
    """Cylinder chart: u along the axis e_u, v = arc length around from the crest; n outward at the crest."""

    def __init__(self, center, e_u, e_v, radius_m, width_m=0.06, height_m=0.06, cell_m=0.001):
        self.center = np.asarray(center, float)
        self.e_u = np.asarray(e_u, float)
        self.e_v = np.asarray(e_v, float)
        self.n = np.cross(self.e_u, self.e_v)
        self.radius_m = radius_m
        self.width_m = width_m
        self.height_m = height_m
        rows = int(round(height_m / cell_m)) + 1
        cols = int(round(width_m / cell_m)) + 1
        self.count = np.ones((rows, cols), dtype=np.int32)
        self.chart = SimpleNamespace(kind="cylinder", radius_m=radius_m)

    def frame(self, uv):
        uv = np.asarray(uv, float)
        phi = uv[:, 1] / self.radius_m
        s, c = np.sin(phi)[:, None], np.cos(phi)[:, None]
        point = self.center + uv[:, :1] * self.e_u + self.radius_m * (s * self.e_v + (c - 1.0) * self.n)
        normal = s * self.e_v + c * self.n
        d_dv = c * self.e_v - s * self.n
        return point, np.repeat(self.e_u[None], len(uv), 0), d_dv, normal

    def project(self, points):
        q = np.asarray(points, float) - self.center
        u = q @ self.e_u
        y = q @ self.e_v
        z = q @ self.n + self.radius_m
        phi = np.arctan2(y, z)
        return np.stack([u, self.radius_m * phi], axis=1), np.hypot(y, z) - self.radius_m


def _contact(surface_center_root, normal, lean_deg=0.0):
    """Contact pose in base with the tool axis lean_deg off -normal (rotated about base x)."""
    r_c = dk.align_rotation(TOOL_AXIS, -np.asarray(normal, float))
    if lean_deg:
        r_c = dk.axis_rotation([1.0, 0.0, 0.0], math.radians(lean_deg)) @ r_c
    tip = dk.base_from_root(surface_center_root)
    return {"schema": "tatbot.draw-pose/1", "frame": "right/base_link", "period_s": PERIOD,
            "tip": tip.tolist(), "rotation": r_c.tolist(), "tool": "lutin-ballpoint-dot"}


def _hold(contact, normal, standoff_m=0.12):
    tip = np.asarray(contact["tip"]) + standoff_m * np.asarray(normal)
    return dict(contact, tip=tip.tolist())


CENTER_ROOT = np.array([0.35, -0.05, 0.02])


# --- time law and spiral --------------------------------------------------------

def test_time_law_totals_and_continuity():
    length = 0.0572
    t, s, sdot = dp.time_law(length, 120.0, 2.0, PERIOD)
    assert len(t) == 48000
    assert t[0] == pytest.approx(PERIOD) and t[-1] == pytest.approx(120.0)
    assert s[-1] == pytest.approx(length, abs=1e-12)
    assert np.all(np.diff(s) >= -1e-15)
    cruise = length / 118.0
    assert sdot.max() == pytest.approx(cruise, rel=1e-12)
    assert sdot[-1] == pytest.approx(0.0, abs=1e-9)
    # speed is continuous at the ease boundaries and the distance integrates it
    assert np.abs(np.diff(sdot)).max() < cruise * 0.01
    assert np.abs(np.diff(s) / PERIOD - 0.5 * (sdot[1:] + sdot[:-1])).max() < cruise * 0.01
    with pytest.raises(ValueError):
        dp.time_law(length, 3.0, 2.0, PERIOD)


def test_spiral_polyline_matches_the_closed_form():
    radius, turns = 0.006, 3.0
    poly = dp.spiral_polyline(radius, turns)
    closed = dp.spiral_path_length(radius, turns)
    assert abs(dp.polyline_length(poly) - closed) / closed < 1e-3
    assert np.allclose(poly[0], 0.0)
    assert np.allclose(poly[-1], [radius, 0.0], atol=1e-12)
    assert closed == pytest.approx(0.05722, abs=5e-5)


def test_resample_by_arclength_hits_vertices():
    poly = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 2.0]])
    points, tangents = dp.resample_polyline_by_arclength(poly, np.array([0.0, 0.5, 1.0, 2.0, 3.0]))
    assert np.allclose(points, [[0, 0], [0.5, 0], [1, 0], [1, 1], [1, 2]])
    assert np.allclose(tangents[1], [1, 0]) and np.allclose(tangents[-1], [0, 1])


# --- compile on surfaces ----------------------------------------------------------

def test_compile_path_on_a_plane_lands_on_the_surface():
    plane = FakePlane(CENTER_ROOT, [1, 0, 0], [0, 1, 0])
    contact = _contact(CENTER_ROOT, plane.n)
    hold = _hold(contact, plane.n)
    samples, report = dp.compile_path(plane, CONFIG, contact, hold, PERIOD)
    down = samples.pen > 0
    assert down.sum() == 12000
    _, dist = plane.project(dk.root_from_base(samples.p[down]))
    assert np.abs(dist).max() < 1e-6
    axes = np.einsum("nij,j->ni", samples.R[down], TOOL_AXIS)
    assert np.allclose(axes, -plane.n, atol=1e-9)
    assert np.allclose(samples.p[0], hold["tip"]) and np.allclose(samples.R[0], hold["rotation"])
    assert report["anchor_gap_mm"] < 1e-6
    # pen-down starts at the anchor and ends at radius along +u
    assert np.linalg.norm(samples.p[down][0] - contact["tip"]) < 5e-5
    assert np.allclose(samples.p[down][-1], np.asarray(contact["tip"]) + [0.006, 0, 0], atol=1e-9)
    # pen-up before, pen-up after; approach speed obeys the caps
    assert samples.pen[0] == 0 and samples.pen[-1] == 0
    assert np.linalg.norm(samples.v, axis=1).max() <= dp.APPROACH_SPEED_M_S + 1e-4
    pre = dp.preflight(samples, plane, CONFIG, TOOL_AXIS, hold, design_length_m=report["design_length_mm"] * 1e-3)
    assert pre["lean_max_deg"] < 1e-6
    assert pre["holes"] == 0
    assert abs(pre["arc_length_ratio"] - 1.0) < 2e-3
    assert pre["normal_swing_max_deg"] < 1e-9


def test_compile_path_on_a_40mm_cylinder_keeps_arc_length_and_normal():
    cylinder = FakeCylinder(CENTER_ROOT, [1, 0, 0], [0, 1, 0], radius_m=0.04)
    contact = _contact(CENTER_ROOT, cylinder.n)
    hold = _hold(contact, cylinder.n)
    samples, report = dp.compile_path(cylinder, CONFIG, contact, hold, PERIOD)
    pre = dp.preflight(samples, cylinder, CONFIG, TOOL_AXIS, hold, design_length_m=report["design_length_mm"] * 1e-3)
    assert abs(pre["arc_length_ratio"] - 1.0) < 5e-3
    assert pre["lean_max_deg"] < 0.1
    assert pre["chart_kind"] == "cylinder"
    assert 7.0 < pre["normal_swing_max_deg"] < 9.0   # 5.5 mm of v-extent around a 40 mm radius
    down = samples.pen > 0
    _, dist = cylinder.project(dk.root_from_base(samples.p[down]))
    assert np.abs(dist).max() < 1e-6


def test_preflight_refuses_lean_holes_and_start():
    plane = FakePlane(CENTER_ROOT, [1, 0, 0], [0, 1, 0])
    contact = _contact(CENTER_ROOT, plane.n, lean_deg=25.0)
    hold = _hold(contact, plane.n)
    samples, _ = dp.compile_path(plane, CONFIG, contact, hold, PERIOD)
    with pytest.raises(dp.DrawRefusal) as info:
        dp.preflight(samples, plane, CONFIG, TOOL_AXIS, hold)
    assert info.value.code == "lean_over_budget"

    contact = _contact(CENTER_ROOT, plane.n)
    hold = _hold(contact, plane.n)
    samples, _ = dp.compile_path(plane, CONFIG, contact, hold, PERIOD)
    holed = FakePlane(CENTER_ROOT, [1, 0, 0], [0, 1, 0])
    rows, cols = holed.count.shape
    holed.count[rows // 2, cols // 2 + 2] = 0     # the spiral crosses +u at r = 2 mm (theta = 2 pi)
    with pytest.raises(dp.DrawRefusal) as info:
        dp.preflight(samples, holed, CONFIG, TOOL_AXIS, hold)
    assert info.value.code == "holes"

    off = dict(hold, tip=(np.asarray(hold["tip"]) + [0.005, 0, 0]).tolist())
    with pytest.raises(dp.DrawRefusal) as info:
        dp.preflight(samples, plane, CONFIG, TOOL_AXIS, off)
    assert info.value.code == "start_tolerance"

    fast = dp.Samples(PERIOD, samples.t, samples.p, samples.v * 5.0, samples.R, samples.pen, samples.capture)
    with pytest.raises(dp.DrawRefusal) as info:
        dp.preflight(fast, plane, CONFIG, TOOL_AXIS, hold)
    assert info.value.code == "tip_speed"


def test_preflight_refuses_girth_on_a_small_cylinder():
    cylinder = FakeCylinder(CENTER_ROOT, [1, 0, 0], [0, 1, 0], radius_m=0.003, width_m=0.06, height_m=0.06)
    contact = _contact(CENTER_ROOT, cylinder.n)
    hold = _hold(contact, cylinder.n)
    samples, _ = dp.compile_path(cylinder, CONFIG, contact, hold, PERIOD)
    with pytest.raises(dp.DrawRefusal) as info:
        dp.preflight(samples, cylinder, CONFIG, TOOL_AXIS, hold)
    assert info.value.code == "girth"


def test_transported_rotations_carry_the_contact_normal():
    rng = np.random.default_rng(5)
    n_c = np.array([0.0, 0.0, 1.0])
    r_c = dk.align_rotation(TOOL_AXIS, -n_c)
    normals = rng.normal(size=(50, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    rotations = dp.transported_rotations(normals, n_c, r_c)
    for i in range(50):
        assert np.allclose(rotations[i] @ rotations[i].T, np.eye(3), atol=1e-12)
        assert np.allclose(rotations[i] @ TOOL_AXIS, -normals[i], atol=1e-12)
        assert np.allclose(rotations[i], dk.align_rotation(n_c, normals[i]) @ r_c, atol=1e-12)


def test_compiled_path_passes_the_executor_caps_from_the_witness_pose():
    """The C++ witness pose is a real contact: stroke, lift and descent must clear plan_joints.

    Pen-up rows are planned with the carriage held (lock_carriage_when_up):
    the 7-DoF loop cannot lift 60 mm along the tool axis at 10 mm/s without
    the carriage leaving its envelope (see draw_kinematics.plan_joints).
    """
    witness = np.array([0.173762112856, 1.544403791428, 0.826848268509,
                        -0.061226826161, 0.121118485928, 1.642061471939])
    tip_w, r_w, _ = dk.fk_ballpoint(witness, dk.CARRIAGE_IK_BIAS_M)
    normal = -(r_w @ TOOL_AXIS)
    e_u = np.array([1.0, 0.0, 0.0]) - np.dot([1.0, 0.0, 0.0], normal) * normal
    e_u /= np.linalg.norm(e_u)
    plane = FakePlane(dk.root_from_base(tip_w), e_u, np.cross(normal, e_u))
    contact = dict(_contact(plane.center, plane.n), tip=tip_w.tolist(), rotation=r_w.tolist())
    config = dict(CONFIG, design={"kind": "spiral", "radius_mm": 3, "turns": 2, "rotation_deg": 0},
                  duration_s=40, orbit=dict(CONFIG["orbit"], standoff_mm=60))
    hold = _hold(contact, plane.n, standoff_m=0.06)
    samples, report = dp.compile_path(plane, config, contact, hold, PERIOD)
    first_down = int(np.argmax(samples.pen > 0))
    last_down = int(len(samples.pen) - np.argmax(samples.pen[::-1] > 0) - 1)

    def part(a, b):
        return {"p": samples.p[a:b], "v": samples.v[a:b], "R": samples.R[a:b], "pen": samples.pen[a:b]}

    stroke = dk.plan_joints(part(first_down, last_down + 1), witness, dk.CARRIAGE_IK_BIAS_M, PERIOD)
    stats = stroke["stats"]
    assert stats["max_model_error_mm"] < 0.01
    assert stats["min_carriage_m"] >= 0.0008 and stats["max_carriage_m"] <= 0.0032
    assert stats["max_carriage_acceleration_m_s2"] < 0.5 * dk.PLAN_MAX_CARRIAGE_ACCELERATION_M_S2
    # the 7-DoF loop refuses the lift; with the carriage held on pen-up rows it passes
    end = stroke["positions"][-1]
    with pytest.raises(dk.PlanRefusal):
        dk.plan_joints(part(last_down + 1, samples.n), end[:6], end[6], PERIOD,
                       stats["end_carriage_velocity_m_s"])
    lift = dk.plan_joints(part(last_down + 1, samples.n), end[:6], end[6], PERIOD,
                          stats["end_carriage_velocity_m_s"], lock_carriage_when_up=True)
    assert lift["stats"]["max_model_error_mm"] < 0.1
    # the carriage ramps to rest from the stroke's centering velocity, then holds
    assert lift["stats"]["max_carriage_velocity_m_s"] <= abs(stats["end_carriage_velocity_m_s"]) + 1e-12
    assert abs(lift["positions"][-1, 6] - end[6]) < 1e-5
    # travel from above the stroke end to above the anchor, then the compiled descent
    top = lift["positions"][-1]
    standoff_pt = samples.p[report["approach_ticks"] - 1]
    travel_p, travel_r = dp.line_segment(samples.p[-1], standoff_pt, samples.R[-1], samples.R[-1], 0.01, PERIOD)
    travel = dk.plan_joints({"p": travel_p, "v": dp.feedforward(travel_p, PERIOD), "R": travel_r,
                             "pen": np.zeros(len(travel_p), int)}, top[:6], top[6], PERIOD,
                            lock_carriage_when_up=True)
    above = travel["positions"][-1]
    descent = dk.plan_joints(part(report["approach_ticks"], first_down), above[:6], above[6], PERIOD,
                             lock_carriage_when_up=True)
    assert descent["stats"]["max_model_error_mm"] < 0.1
    assert descent["stats"]["endpoint_error_m"] < 1e-4


def test_draw_speed_sets_the_cruise_and_the_duration():
    """draw_speed_mm_s overrides duration_s: cruise = speed, stroke duration = length / speed + ease."""
    plane = FakePlane(CENTER_ROOT, [1, 0, 0], [0, 1, 0])
    contact = _contact(CENTER_ROOT, plane.n)
    hold = _hold(contact, plane.n)
    config = dict(CONFIG, design={"kind": "spiral", "radius_mm": 15, "turns": 3, "rotation_deg": 0},
                  draw_speed_mm_s=3.5, duration_s=999)
    samples, report = dp.compile_path(plane, config, contact, hold, PERIOD)
    stroke = report["strokes"][0]
    assert abs(stroke["cruise_speed_mm_s"] - 3.5) < 0.02
    assert abs(stroke["duration_s"] - (stroke["length_mm"] / 3.5 + CONFIG["ease_s"])) < 1e-9
    assert 140 < stroke["length_mm"] < 146 and 40 < stroke["duration_s"] < 45
    assert report["draw_speed_mm_s"] == 3.5
    down = samples.pen > 0
    assert abs(np.linalg.norm(samples.v[down], axis=1).max() - 0.0035) < 5e-5
    # too slow for the ease is a refusal, not a clamp
    with pytest.raises(dp.DrawRefusal) as info:
        dp.compile_path(plane, dict(config, design={"kind": "spiral", "radius_mm": 2, "turns": 1, "rotation_deg": 0}),
                        contact, hold, PERIOD)
    assert info.value.code == "design"
    # without a speed the duration is honoured as before
    samples2, report2 = dp.compile_path(plane, dict(config, draw_speed_mm_s=None, duration_s=60), contact, hold, PERIOD)
    assert report2["draw_speed_mm_s"] is None and abs(report2["strokes"][0]["duration_s"] - 60) < 1e-9


def test_approach_is_short_fast_then_gentle():
    """hold -> 30 mm above the anchor at 20 mm/s, 10 mm/s down to 5 mm, 3 mm/s to the contact, settle."""
    plane = FakePlane(CENTER_ROOT, [1, 0, 0], [0, 1, 0])
    contact = _contact(CENTER_ROOT, plane.n)
    hold = _hold(contact, plane.n)
    samples, report = dp.compile_path(plane, CONFIG, contact, hold, PERIOD)
    assert report["approach_mm"] == dp.APPROACH_STANDOFF_M * 1e3
    above = samples.p[report["approach_ticks"] - 1]
    assert abs(np.dot(above - contact["tip"], plane.n) - dp.APPROACH_STANDOFF_M) < 1e-9
    assert abs(np.linalg.norm(above - contact["tip"]) - dp.APPROACH_STANDOFF_M) < 1e-9
    first_down = int(np.argmax(samples.pen > 0))
    height = (samples.p[:first_down] - np.asarray(contact["tip"])) @ plane.n
    speed = np.linalg.norm(samples.v[:first_down], axis=1)
    assert speed[height > dp.APPROACH_STANDOFF_M + 1e-6].max() <= dp.APPROACH_SPEED_M_S + 1e-4
    mid = (height < dp.APPROACH_STANDOFF_M - 1e-6) & (height > dp.FINAL_DESCENT_M + 1e-6)
    assert speed[mid].max() <= dp.DESCENT_SPEED_M_S + 1e-4 and speed[mid].max() > 0.5 * dp.DESCENT_SPEED_M_S
    last = (height < dp.FINAL_DESCENT_M - 1e-6) & (height > 1e-6)
    assert speed[last].max() <= dp.FINAL_DESCENT_SPEED_M_S + 1e-4
    # the settle before pen-down is still there, and the whole pen-up head is under 20 s
    settle = int(round(dp.DESCENT_SETTLE_S / PERIOD))
    assert np.max(np.abs(samples.p[first_down - settle:first_down] - samples.p[first_down - 1])) < 1e-12
    assert samples.t[first_down] < 20.0
    # the lift after the stroke rises the approach standoff, not the orbit standoff
    assert abs(np.dot(samples.p[-1] - samples.p[samples.pen > 0][-1], plane.n) - dp.APPROACH_STANDOFF_M) < 1e-9


def test_lean_deadband_shortens_the_transported_rotation():
    """Inside the deadband the rotation is the contact's; outside it is shortened by the deadband."""
    n_c = np.array([0.0, 0.0, 1.0])
    r_c = dk.align_rotation(TOOL_AXIS, -n_c)
    swing = np.radians(np.array([0.0, 5.0, 12.0, 19.0]))
    normals = np.stack([[math.sin(a), 0.0, math.cos(a)] for a in swing])
    full = dp.transported_rotations(normals, n_c, r_c)
    relaxed = dp.transported_rotations(normals, n_c, r_c, deadband_rad=math.radians(12.0))
    lean_full = np.degrees([math.acos(np.clip(-(r @ TOOL_AXIS) @ n, -1, 1)) for r, n in zip(full, normals, strict=True)])
    lean_relaxed = np.degrees([math.acos(np.clip(-(r @ TOOL_AXIS) @ n, -1, 1)) for r, n in zip(relaxed, normals, strict=True)])
    assert np.allclose(lean_full, 0.0, atol=1e-9)
    assert np.allclose(lean_relaxed, [0.0, 5.0, 12.0, 12.0], atol=1e-9)
    assert np.allclose(relaxed[0], r_c) and np.allclose(relaxed[1], r_c) and np.allclose(relaxed[2], r_c, atol=1e-12)
    assert np.allclose(dp.transported_rotations(normals, n_c, r_c, deadband_rad=0.0), full)
    # the compiled path honours the config key and reports it
    cylinder = FakeCylinder(CENTER_ROOT, [1, 0, 0], [0, 1, 0], 0.04)
    contact = _contact(cylinder.frame(np.zeros((1, 2)))[0][0], cylinder.frame(np.zeros((1, 2)))[3][0])
    hold = _hold(contact, cylinder.frame(np.zeros((1, 2)))[3][0])
    config = dict(CONFIG, design={"kind": "spiral", "radius_mm": 15, "turns": 3, "rotation_deg": 0},
                  draw_speed_mm_s=3.5, lean_deadband_deg=12)
    samples, report = dp.compile_path(cylinder, config, contact, hold, PERIOD)
    pre = dp.preflight(samples, cylinder, config, TOOL_AXIS, hold)
    assert abs(report["lean_deadband_deg"] - 12) < 1e-9
    assert 11.0 < pre["lean_max_deg"] <= 12.0 + 1e-6 and pre["normal_swing_max_deg"] > 18.0


def test_design_patch_ring_follows_the_design():
    assert dp.design_patch_ring_m(None) == dp.PATCH_RING_M
    assert dp.design_patch_ring_m({"kind": "spiral", "radius_mm": 3, "turns": 2, "rotation_deg": 0}) == dp.PATCH_RING_M
    assert abs(dp.design_patch_ring_m({"kind": "spiral", "radius_mm": 15, "turns": 3, "rotation_deg": 0}) - 0.018) < 1e-6
    assert dp.design_patch_ring_m({"kind": "unknown"}) == dp.PATCH_RING_M


# A recorded curved-target contact pose.
BOTTLE_TRIGGER_JOINTS = np.array([0.21915769577, 0.73338675499, 0.269512474537, 0.082589454949, -0.071526661515, 1.59285116196])


def test_camera_orbit_backs_off_the_off_axis_angle_for_a_wide_design():
    """A 15 mm design needs an 18 mm ring in both frustums: 35 deg cannot, 30 deg can, from the real touch."""
    tip_c, r_c, _ = dk.fk_ballpoint(BOTTLE_TRIGGER_JOINTS, dk.CARRIAGE_IK_BIAS_M)
    trigger = {"schema": "tatbot.draw-pose/1", "frame": "right/base_link", "period_s": PERIOD,
               "tip": tip_c.tolist(), "rotation": r_c.tolist(), "tool": "lutin-ballpoint-dot",
               "joints": BOTTLE_TRIGGER_JOINTS.tolist(), "carriage_m": dk.CARRIAGE_IK_BIAS_M}
    config = json.loads(json.dumps(CONFIG))
    config["orbit"] = {"mode": "camera", "camera_distance_mm": 160, "off_axis_deg": 35, "tilt_deg": 15,
                       "poses": 5, "speed_mm_s": 20.0}
    config["design"] = {"kind": "spiral", "radius_mm": 6, "turns": 3, "rotation_deg": 0}
    _, small = dp.orbit_samples(config, trigger, PERIOD, TOOL_AXIS)
    assert small["off_axis_deg"] == 35 and abs(small["patch_ring_mm"] - 9) < 1e-6 and small["speed_factor"] == 1.0
    config["design"]["radius_mm"] = 15
    samples, wide = dp.orbit_samples(config, trigger, PERIOD, TOOL_AXIS)
    assert wide["off_axis_config_deg"] == 35 and abs(wide["off_axis_deg"] - 30) < 1e-9
    assert abs(wide["patch_ring_mm"] - 18) < 1e-6 and wide["speed_factor"] == 1.0
    assert all(c["in_view"] >= 1.0 for s in wide["viewpoint_scores"] for c in s["cameras"].values())
    assert abs(wide["duration_s"] - small["duration_s"]) < 3.0
    # the widest design the CLI allows backs off one more step from this touch
    config["design"]["radius_mm"] = 30
    config["map"]["extent_mm"] = 100
    _, widest = dp.orbit_samples(config, trigger, PERIOD, TOOL_AXIS)
    assert abs(widest["patch_ring_mm"] - 33) < 1e-6 and abs(widest["off_axis_deg"] - 25) < 1e-9
    assert widest["speed_factor"] == 1.0


# --- files ------------------------------------------------------------------------

def test_samples_csv_round_trips(tmp_path):
    plane = FakePlane(CENTER_ROOT, [1, 0, 0], [0, 1, 0])
    contact = _contact(CENTER_ROOT, plane.n)
    hold = _hold(contact, plane.n)
    samples, _ = dp.compile_path(plane, CONFIG, contact, hold, PERIOD)
    path = tmp_path / "path.csv"
    dp.write_samples_csv(path, samples, "path", dk.BALLPOINT_TIP_IN_LINK6,
                         {"lean_max_deg": 0.0, "path_length_mm": 57.22, "note": "test"})
    text = path.read_text().splitlines()
    assert text[0] == "schema,tatbot.draw-samples/1"
    assert text[1] == "path,path".replace("path,path", "kind,path")
    assert text[2] == "frame,right/base_link"
    assert text[3] == "period_s,0.0025"
    assert text[4:7] == ["tip_x_m,0.20550927", "tip_y_m,0.01083364", "tip_z_m,-0.00149001"]
    assert text[7] == f"sample_count,{samples.n}"
    assert text[8] == "start_tolerance_m,0.001"
    assert "columns," + ",".join(dp.COLUMNS) in text
    assert text[-1].split(",")[0] == format(samples.t[-1], ".12g")
    back, header = dp.read_samples_csv(path)
    assert header["kind"] == "path" and header["note"] == "test" and header["lean_max_deg"] == 0.0
    assert back.n == samples.n
    assert np.allclose(back.p, samples.p, atol=1e-12, rtol=1e-11)
    assert np.allclose(back.v, samples.v, atol=1e-12, rtol=1e-11)
    assert np.allclose(back.R, samples.R, atol=1e-12, rtol=1e-11)
    assert np.array_equal(back.pen, samples.pen) and np.array_equal(back.capture, samples.capture)
    with pytest.raises(ValueError):
        dp.write_samples_csv(path, samples, "path", dk.BALLPOINT_TIP_IN_LINK6, {"bad,key": 1})


# --- orbit --------------------------------------------------------------------------

def test_orbit_samples_capture_rows_and_standoff():
    normal = np.array([0.0, 0.0, 1.0])
    trigger = _contact(CENTER_ROOT, normal)
    tip_config = json.loads(json.dumps(CONFIG))
    tip_config.setdefault("orbit", {})["mode"] = "tip"
    tip_config["orbit"]["standoff_mm"] = 120
    samples, report = dp.orbit_samples(tip_config, trigger, PERIOD, TOOL_AXIS)
    assert report["capture_count"] == 5
    assert np.all(samples.pen == 0)
    captures = samples.capture[samples.capture > 0]
    assert captures.tolist() == [1, 2, 3, 4, 5]
    assert samples.capture[-1] == 5
    assert np.allclose(samples.p[0], trigger["tip"])
    lift_ticks = dp._segment_ticks(0.12, 0.01, PERIOD) + 1
    dist = np.linalg.norm(samples.p[lift_ticks:] - np.asarray(trigger["tip"]), axis=1)
    assert np.abs(dist - 0.12).max() < 1e-3
    # each capture row is stationary for the 0.5 s hold before it
    hold_ticks = int(round(0.5 / PERIOD))
    for row in np.flatnonzero(samples.capture > 0):
        block = samples.p[row - hold_ticks + 1:row + 1]
        assert np.abs(block - block[0]).max() < 1e-12
    # viewpoint 1 is straight above; the tilted ones are 15 deg off
    views = np.asarray(report["viewpoints_base"]) - np.asarray(trigger["tip"])
    angles = np.degrees(np.arccos(np.clip(views @ normal / np.linalg.norm(views, axis=1), -1, 1)))
    assert np.allclose(angles, [0, 15, 15, 15, 15], atol=1e-9)
    pre = dp.preflight(samples, None, tip_config, TOOL_AXIS, trigger)
    assert pre["tip_speed_max_mm_s"] <= 10.1
    dp.write_samples_csv(Path("/dev/null"), samples, "orbit", dk.BALLPOINT_TIP_IN_LINK6, report_scalars(report))


def report_scalars(report):
    return {k: v for k, v in report.items() if isinstance(v, (int, float, str)) and not isinstance(v, bool)}


def test_orbit_three_poses():
    trigger = _contact(CENTER_ROOT, [0.0, 0.0, 1.0])
    config = dict(CONFIG, orbit=dict(CONFIG["orbit"], poses=3))
    samples, report = dp.orbit_samples(config, trigger, PERIOD, TOOL_AXIS)
    assert samples.capture[samples.capture > 0].tolist() == [1, 2, 3]
    assert report["capture_count"] == 3


def test_camera_orbit_puts_both_cameras_on_the_patch_with_the_tip_clear():
    """Decision: camera-centric orbit. Cameras ~160 mm from the contact looking at it, patch in both
    frustums at every viewpoint, tip well up and to the side, samples start at the contact."""
    normal = np.array([0.0, 0.0, 1.0])
    trigger = _contact(CENTER_ROOT, normal)
    # the contact rotation must be a real tool pose: tool axis along -normal with the rig axes
    # from the URDF, so build it from the witness joints' rotation aligned onto the normal
    r_w = dk.fk_ballpoint(np.array([0.1738, 1.5444, 0.8268, -0.0612, 0.1211, 1.6421]), 0.002)[1]
    r_c = dk.align_rotation(r_w @ TOOL_AXIS, -normal) @ r_w
    trigger["rotation"] = r_c.tolist()
    trigger["carriage_m"] = 0.002
    config = json.loads(json.dumps(CONFIG))
    config["orbit"] = {"mode": "camera", "camera_distance_mm": 160, "off_axis_deg": 35, "tilt_deg": 15,
                       "poses": 5, "speed_mm_s": 10}
    samples, report = dp.orbit_samples(config, trigger, PERIOD, TOOL_AXIS)
    assert report["mode"] == "camera" and report["capture_count"] == 5
    assert np.allclose(samples.p[0], trigger["tip"])
    assert samples.capture[samples.capture > 0].tolist() == [1, 2, 3, 4, 5]
    tip_c = np.asarray(trigger["tip"])
    for score in report["viewpoint_scores"]:
        for cam in score["cameras"].values():
            assert cam["in_view"] == 1.0
            assert 130.0 < cam["distance_mm"] < 200.0, cam
            assert cam["incidence_deg"] < 45.0, cam
        assert 25.0 < score["tip_height_mm"] < 70.0, score
        assert 60.0 < score["tip_offset_mm"] < 110.0, score
    # the mean camera really is camera_distance from the contact at every capture row
    for row in np.flatnonzero(samples.capture > 0):
        cams = dk.rig_cameras(samples.p[row], samples.R[row], 0.002)
        mean = np.mean([c[:3, 3] for c in cams.values()], axis=0)
        assert abs(np.linalg.norm(mean - tip_c) - 0.160) < 1e-6
    pre = dp.preflight(samples, None, config, TOOL_AXIS, trigger)
    assert pre["tip_speed_max_mm_s"] <= 10.0 + 1e-6


def test_camera_orbit_adaptively_backs_off_speed_on_joint_velocity():
    """fails if camera orbit speed backoff is skipped or miscalculates the reported speed factor when joint velocity is exceeded."""
    witness = np.array([0.173762112856, 1.544403791428, 0.826848268509,
                        -0.061226826161, 0.121118485928, 1.642061471939])
    tip_w, r_w, _ = dk.fk_ballpoint(witness, 0.002)
    trigger = {
        "schema": "tatbot.draw-pose/1", "frame": "right/base_link", "period_s": PERIOD,
        "tip": tip_w.tolist(), "rotation": r_w.tolist(), "tool": "lutin-ballpoint-dot",
        "joints": witness.tolist(), "carriage_m": 0.002,
    }
    config = json.loads(json.dumps(CONFIG))
    config["orbit"] = {
        "mode": "camera", "camera_distance_mm": 160, "off_axis_deg": 35, "tilt_deg": 15,
        "poses": 5, "speed_mm_s": 20.0, "rotation_deg_s": 12.0,
    }
    samples, report = dp.orbit_samples(config, trigger, PERIOD, TOOL_AXIS)
    assert report["speed_factor"] < 1.0
    assert report["speed_mm_s"] == pytest.approx(20.0 * report["speed_factor"])
    assert any("joint_velocity" in s for s in report["sides_refused"])
    assert samples.n > 0


def test_compile_path_holds_stationary_at_contact_before_pen_down():
    """fails if the compiled path omits the stationary settle hold before pen-down contact."""
    plane = FakePlane(CENTER_ROOT, [1, 0, 0], [0, 1, 0])
    contact = _contact(CENTER_ROOT, plane.n)
    hold = _hold(contact, plane.n)
    samples, _ = dp.compile_path(plane, CONFIG, contact, hold, PERIOD)
    first_down = int(np.argmax(samples.pen > 0))
    settle_ticks = int(round(dp.DESCENT_SETTLE_S / PERIOD))
    settle_p = samples.p[first_down - settle_ticks:first_down]
    settle_pen = samples.pen[first_down - settle_ticks:first_down]
    assert np.all(settle_pen == 0)
    assert np.max(np.abs(settle_p - settle_p[0])) < 1e-12
