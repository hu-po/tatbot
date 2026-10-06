"""The wrist gauge's fit (tatbot_session.gauge): one pen tip fixed in the camera's frame and frames whose paper
planes turn by the tool's +-2 deg, as the trips of 2026-10-03 did; the contacts fit to zero, the trips in the air
read their own heights."""
from __future__ import annotations

import numpy as np
import pytest
from tatbot_session import gauge


def _turned(n, deg):
    axis = np.cross(n, [1.0, 0.0, 0.0])
    axis /= np.linalg.norm(axis)
    a = np.radians(deg)
    return n * np.cos(a) + np.cross(axis, n) * np.sin(a) + axis * (axis @ n) * (1 - np.cos(a))


def test_the_fit_puts_the_contacts_at_zero_and_reads_the_air_trips():
    rng = np.random.default_rng(0)
    n0 = np.array([-0.727, 0.243, -0.642])
    n0 /= np.linalg.norm(n0)
    end = np.array([-0.018, 0.007, 0.160])        # the cone's end in the camera frame
    pen = end + np.outer(rng.uniform(0.0, 0.03, 4000), n0) + rng.normal(0.0, 0.0015, (4000, 3)) * [1, 1, 0]
    truth = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.003, 0.006, 0.010, 0.016]
    frames = []
    for k, h in enumerate(truth):
        n = _turned(n0, 2.0 if k % 2 else -2.0)
        lowest = pen[np.argmin(pen @ n0)]
        c = lowest - (h + 0.0025) * n + rng.normal(0.0, 0.00005, 3)   # the cone's end 2.5 mm over a contact
        frames.append(gauge.Frame(pen + rng.normal(0.0, 0.0002, pen.shape), c, n, 0.0004))
    cal, heights = gauge.fit(frames)
    assert cal["contacts"] == 6
    np.testing.assert_allclose(heights, truth, atol=4e-4)
    assert gauge.height(frames[8], cal) == pytest.approx(0.010, abs=4e-4)


def test_a_fitted_gauge_finds_a_pen_of_any_colour_on_its_axis():
    h, w, f = 480, 640, 400.0
    v, u = np.mgrid[0:h, 0:w]
    z = np.full((h, w), 0.200)                          # the paper, square to the camera
    pen = (np.abs(u - 320) <= 20) & (v >= 240) & (v <= 400)
    z[pen] = 0.12 + (v[pen] - 240) * 0.0002             # a pink pen leaning toward the paper
    colour = np.full((h, w, 3), 230, np.uint8)
    colour[pen] = (140, 65, 235)
    meta = {"depth_units_m": 0.0001, "intrinsics": {"fx": f, "fy": f, "ppx": 320.0, "ppy": 240.0},
            "color_from_depth": {"rotation": np.eye(3).ravel().tolist(), "translation_m": [0.0, 0.0, 0.0]},
            "color_intrinsics": {"k": [[f, 0.0, 320.0], [0.0, f, 240.0], [0.0, 0.0, 1.0]]}}
    raw = np.round(z / 0.0001).astype(np.uint16)
    pts = np.stack([(u[pen] - 320) / f * z[pen], (v[pen] - 240) / f * z[pen], z[pen]], axis=1)
    m, a = gauge._axis(pts)
    a = a if a[2] > 0 else -a
    cal = {"cone_end_cam_m": (m + np.percentile((pts - m) @ a, 99.5) * a).tolist(), "axis_cam": a.tolist()}
    assert gauge.measure(raw, colour, meta) is None     # not teal: the colour finds no pen
    frame = gauge.measure(raw, colour, meta, cal)
    assert frame is not None and len(frame.pen) >= 100
    np.testing.assert_allclose(frame.n, [0.0, 0.0, -1.0], atol=1e-3)
    assert frame.c[2] == pytest.approx(0.200, abs=2e-4)
    assert np.max(gauge._along(frame.pen, cal)[1]) < gauge.AXIS_M


def test_pink_silicone_is_a_surface_beside_a_print_standing_off_it():
    h, w, f = 480, 640, 400.0
    v, u = np.mgrid[0:h, 0:w]
    z = np.full((h, w), 0.200)
    pen = (np.abs(u - 320) <= 20) & (v >= 240) & (v <= 400)
    z[pen] = 0.12 + (v[pen] - 240) * 0.0002
    colour = np.full((h, w, 3), (166, 181, 232), np.uint8)   # practice skin's peach (BGR), S ~73
    colour[pen] = (200, 160, 30)                             # the teal cartridge
    stripe = (u < 100) & ~pen
    z[stripe] = 0.190                                        # a violet print 10 mm proud of the skin...
    colour[stripe] = (200, 40, 120)                          # ...too saturated to be surface
    meta = {"depth_units_m": 0.0001, "intrinsics": {"fx": f, "fy": f, "ppx": 320.0, "ppy": 240.0},
            "color_from_depth": {"rotation": np.eye(3).ravel().tolist(), "translation_m": [0.0, 0.0, 0.0]},
            "color_intrinsics": {"k": [[f, 0.0, 320.0], [0.0, f, 240.0], [0.0, 0.0, 1.0]]}}
    frame = gauge.measure(np.round(z / 0.0001).astype(np.uint16), colour, meta)
    assert frame is not None and frame.c[2] == pytest.approx(0.200, abs=2e-4)


def test_only_what_the_depth_puts_on_the_paper_is_paper():
    h, w, f = 120, 160, 100.0
    z = np.full((h, w), 0.200)
    z[40:60, 70:90] = 0.150                              # a black pen 50 mm over the paper
    z[80:90, 20:30] = 0.0                                # and a hole in the depth
    meta = {"depth_units_m": 0.0001, "intrinsics": {"fx": f, "fy": f, "ppx": 80.0, "ppy": 60.0},
            "color_from_depth": {"rotation": np.eye(3).ravel().tolist(), "translation_m": [0.0, 0.0, 0.0]},
            "color_intrinsics": {"k": [[f, 0.0, 80.0], [0.0, f, 60.0], [0.0, 0.0, 1.0]]}}
    paper = gauge.Frame(np.zeros((1, 3)), np.array([0.0, 0.0, 0.200]), np.array([0.0, 0.0, -1.0]), 0.0)
    seen = gauge.on_paper(np.round(z / 0.0001).astype(np.uint16), meta, paper, (h, w, 3), close_px=3, trim_px=3)
    assert seen[10, 10] and seen[50, 110] and not seen[50, 80] and not seen[40, 70] and not seen[85, 25]
