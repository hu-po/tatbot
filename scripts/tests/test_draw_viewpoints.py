"""draw_viewpoints: frustum scoring and the analytic gap/lean against a chart."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "lib"))
sys.path.insert(0, str(REPO / "scripts" / "vision"))

import draw_surface as ds  # noqa: E402
import draw_viewpoints as dv  # noqa: E402


def test_camera_view_sees_a_patch_ahead_and_not_behind():
    pts = np.array([[0.0, 0.0, 0.2], [0.01, 0.0, 0.2], [-0.01, 0.0, 0.2]])
    normals = np.tile([0.0, 0.0, -1.0], (3, 1))
    frac, dist, inc = dv.camera_view(np.eye(4), pts, normals)
    assert frac == 1.0 and abs(dist - 0.2) < 1e-9 and inc < 3.0
    behind = np.eye(4)
    behind[:3, 3] = [0.0, 0.0, 0.4]
    frac, _, _ = dv.camera_view(behind, pts, normals)
    assert frac == 0.0
    far = np.eye(4)
    far[:3, 3] = [0.0, 0.0, -0.4]  # 0.6 m away: past the depth range
    frac, _, _ = dv.camera_view(far, pts, normals)
    assert frac == 0.0


def test_gap_and_lean_against_a_cylinder_chart():
    r = 0.045
    chart = ds.CylinderChart(np.array([0.0, 0.0, 0.0]), np.eye(3), r)
    surface = ds.HeightFieldSurface(chart, np.zeros((5, 5)), 0.06, 0.06)
    tip = np.array([0.0, 0.0, 0.02])  # 20 mm above the crest
    gap, lean, uv = dv.gap_and_lean(surface, tip, np.array([0.0, 0.0, -1.0]))
    assert abs(gap - 0.02) < 1e-9 and lean < 1e-6 and np.allclose(uv, 0.0, atol=1e-9)
    theta = 0.3
    side = np.array([0.0, (r + 0.01) * np.sin(theta), (r + 0.01) * np.cos(theta) - r])
    gap, lean, uv = dv.gap_and_lean(surface, side, np.array([0.0, 0.0, -1.0]))
    assert abs(gap - 0.01) < 1e-9
    assert abs(lean - np.degrees(theta)) < 1e-6
    assert abs(uv[1] - r * theta) < 1e-9
