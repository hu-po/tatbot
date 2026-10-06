"""stencil_tip: the print's pose from a rendered wrist view of the coded band's knots, and the tip on the print."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("scipy")
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/lib"), str(ROOT / "scripts/vision")]
import stencil_tip as st  # noqa: E402

SETTINGS = {"width_mm": 100.0, "height_mm": 150.0, "margin_mm": 5.0, "frame_mm": 14.0, "spacing_mm": 6.5,
            "stroke_mm": 0.3, "knot_mm": 2.2, "border_inner_mm": [19.473, 19.812, 82.211, 130.133]}
K = np.array([[650.0, 0, 640.0], [0, 650.0, 360.0], [0, 0, 1]])
DIST = np.zeros(5)


def look_at(eye, target, up=(0.0, 1.0, 0.0)):
    """cam_from_print for an optical camera (z forward, y down) at `eye` looking at `target` (page metres)."""
    z = np.subtract(target, eye)
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    r = np.stack([x, np.cross(z, x), z])
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = r, -r @ np.asarray(eye, float)
    return out


def render(lay, pose):
    img = np.full((720, 1280, 3), 235, np.uint8)
    t = np.linspace(0, 2 * math.pi, 32, endpoint=False)
    ring = np.c_[np.cos(t), np.sin(t)] * lay.knot_m / 2
    for c in lay.knots:
        px, front = st.project(c + ring, pose, K, DIST)
        if front.all():
            cv2.fillPoly(img, [np.round(px * 16).astype(np.int32)], (25, 25, 25), cv2.LINE_AA, shift=4)
    return img


def plane_of(pose):
    n = pose[:3, 2]
    n = n if n[2] > 0 else -n
    return n, float(n @ pose[:3, 3])


@pytest.fixture(scope="module")
def scene():
    lay = st.layout(SETTINGS)
    pose = look_at((-0.045, -0.010, 0.150), (-0.010, 0.0, 0.0))
    return lay, pose, render(lay, pose)


def test_layout_is_seed_free_and_in_the_band(scene):
    lay, _, _ = scene
    assert len(lay.knots) > 100
    x0, y0, x1, y1 = lay.inner
    inside = (lay.knots[:, 0] > x0) & (lay.knots[:, 0] < x1) & (lay.knots[:, 1] > y0) & (lay.knots[:, 1] < y1)
    assert not inside.any()                       # no knot in the clear centre
    assert len(st.lattice_steps(lay.step_m)) == 25


@pytest.mark.parametrize("off", [(0.0015, -0.001), (0.0065 + 0.0012, 0.0), (0.00325, 0.00563 - 0.001)])
def test_fit_recovers_the_print_from_a_prior_off_by_up_to_a_lattice_step(scene, off):
    lay, pose, img = scene
    prior = pose @ np.array([[1, 0, 0, off[0]], [0, 1, 0, off[1]], [0, 0, 1, 0], [0, 0, 0, 1.0]])
    fit = st.fit_print(img, lay, prior, K, DIST, plane=plane_of(pose))
    assert fit is not None
    err = (np.linalg.inv(pose) @ fit.cam_from_print)[:2, 3]
    assert np.linalg.norm(err) < 2e-4
    assert fit.margin >= st.MARGIN_MIN and fit.misplaced == 0


def test_tip_on_print_inverts_the_camera_pose(scene):
    _, pose, _ = scene
    tip_page = np.array([0.005, 0.003, 0.004])
    tip_cam = pose[:3, :3] @ tip_page + pose[:3, 3]
    assert np.allclose(st.tip_on_print(pose, tip_cam), tip_page)


def test_snap_to_plane_keeps_the_ray_and_takes_the_plane(scene):
    _, pose, _ = scene
    tilted = pose @ np.array([[1, 0, 0, 0], [0, math.cos(0.03), -math.sin(0.03), 0], [0, math.sin(0.03), math.cos(0.03), 0.004], [0, 0, 0, 1]])
    snapped = st.snap_to_plane(tilted, *plane_of(pose))
    assert np.allclose(np.abs(snapped[:3, 2] @ pose[:3, 2]), 1.0)
    o = snapped[:3, 3]
    assert np.isclose(plane_of(pose)[0] @ o, plane_of(pose)[1])
    assert np.allclose(np.cross(o / np.linalg.norm(o), tilted[:3, 3] / np.linalg.norm(tilted[:3, 3])), 0, atol=1e-9)
